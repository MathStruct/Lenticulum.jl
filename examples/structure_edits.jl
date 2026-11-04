# Greedy structure search over switch wirings, done as edits of one Mycelium graph and scored by its
# exact free energy (Extending Mycelium §6, L-SPEC-010). Recorded in `examples/structure_edits.md`.
# Run:  julia -t auto --project=docs examples/structure_edits.jl   (an environment with ForwardDiff)
#
# The family is E3's additive wiring with L binary switches: z = c + Σₗ uₗ + ε, uₗ ∈ {bₗ₁, bₗ₂}. The
# graph holds N points, each its own component (data, sum node, L switches), clustered per point
# (`cluster(...; params = :mix)`: the clusters are tied, so every point reads one parameter entry),
# so its free energy is exactly −Σᵢ log p(zᵢ). Two moves, each an edit of the unclustered graph:
# add a switch (every point's sum node gains a channel, `replace_factor!`, and a switch with its
# variables), and remove the least used one. Each candidate is fitted by gradient steps through
# the graph (the free-energy gradient at the exact beliefs, i.e. generalised EM) from a warm start,
# and scored by BIC = 2F + p log N with p = d + L(d + 1) free parameters (a shared offset, one
# direction and one weight per switch). Greedy search starts at L = 0 and takes the best improving
# move until none improves; the reference fits every L ≤ 4 the same way.
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random, Printf, Statistics
using ADTypes, DifferentiationInterface, ForwardDiff


const d = 2
const E = Matrix(1.0I, d, d)
sumfactor(L) = LinearConstraintFactor(NamedTuple{(:z, (Symbol(:u, l) for l in 1:L)...)}(ntuple(_ -> d, L + 1)), d; noise = 0.1)
switch() = SwitchingConstraintFactor((u = d,), 2, d; noise = 1e-3)
nm(s, i) = replica_name(s, i)

# the data graph with no switch: point i has z_i, its data and its sum node
function base_graph(X)
    b = GraphBuilder()
    for i in axes(X, 2)
        variable!(b, nm(:z, i), d)
        factor!(b, nm(:data, i), DataFactor(:z, X[:, i])); connect!(b, nm(:data, i), :z, nm(:z, i); direction = Emitting())
        factor!(b, nm(:sum, i), sumfactor(0); params = :sum, freeze = (:A,)); connect!(b, nm(:sum, i), :z, nm(:z, i))
    end
    return build(b)
end
# the moves, as edits
function add_switch(g, L, N)
    b = GraphBuilder(g)
    for i in 1:N
        replace_factor!(b, nm(:sum, i), sumfactor(L + 1))
        u, k, sw = nm(Symbol(:u, L + 1), i), nm(Symbol(:k, L + 1), i), nm(Symbol(:sw, L + 1), i)
        variable!(b, u, d); variable!(b, k, 2); factor!(b, sw, switch(); params = Symbol(:sw, L + 1), freeze = (:A,))
        connect!(b, nm(:sum, i), Symbol(:u, L + 1), u); connect!(b, sw, :u, u); connect!(b, sw, :k, k)
    end
    return build(b)
end
function remove_switch(g, L, N, l)      # drop switch l, renaming the last one into its place
    b = GraphBuilder(g)
    for i in 1:N
        remove_factor!(b, nm(Symbol(:sw, l), i)); remove_variable!(b, nm(Symbol(:u, l), i)); remove_variable!(b, nm(Symbol(:k, l), i))
    end
    g2 = build(b)
    l == L && return _resum(g2, L - 1, N)
    # rename switch L to l by rebuilding its nodes under the new names
    b = GraphBuilder(g2)
    for i in 1:N
        remove_factor!(b, nm(Symbol(:sw, L), i)); remove_variable!(b, nm(Symbol(:u, L), i)); remove_variable!(b, nm(Symbol(:k, L), i))
        u, k, sw = nm(Symbol(:u, l), i), nm(Symbol(:k, l), i), nm(Symbol(:sw, l), i)
        variable!(b, u, d); variable!(b, k, 2); factor!(b, sw, switch(); params = Symbol(:sw, l), freeze = (:A,))
        connect!(b, sw, :u, u); connect!(b, sw, :k, k)
    end
    return _resum(build(b), L - 1, N)
end
function _resum(g, L, N)
    b = GraphBuilder(g)
    for i in 1:N
        replace_factor!(b, nm(:sum, i), sumfactor(L))
        for l in 1:L
            any(e -> b.factors[e.factor].name === nm(:sum, i) && e.channel === Symbol(:u, l), b.edges) ||
                connect!(b, nm(:sum, i), Symbol(:u, l), nm(Symbol(:u, l), i))
        end
    end
    return build(b)
end
clustered(g, L, N) = cluster(as_plate(g, :pts, N), :pts, :c, [:sum; [Symbol(:sw, l) for l in 1:L]]; params = :mix)   # a plate of clusters

sum_ps(L, c) = (A = NamedTuple{(:z, (Symbol(:u, l) for l in 1:L)...)}((E, ntuple(_ -> -E, L)...)), c = c)
make_ps(L, c, sws) = (; sum = sum_ps(L, c), (Symbol(:sw, l) => sws[l] for l in 1:L)...)
full_ps(gc, inner, N) = merge(NamedTuple{Tuple(nm(:data, i) for i in 1:N)}(ntuple(_ -> NamedTuple(), N)), (mix = inner,))

# Fit by gradient steps at the exact beliefs (Momentum on c, the offsets b and the log-weights; the
# sum node's coefficients stay fixed). Returns the fitted parameters, F and the store.
function fit_gradient(gc, L, N, c, sws; steps = 300, η = 0.02, γ = 0.9, s0 = nothing)
    sched = plate_schedule(gc)                           # N components; two sweeps reach every cluster
    vel = nothing
    s = s0
    F = Inf
    for t in 1:steps
        inner = make_ps(L, c, sws)
        ps = full_ps(gc, inner, N)
        s = s === nothing ? MessageStore(gc) : s
        sweep!(s, gc, sched, ps, NamedTuple()); sweep!(s, gc, sched, ps, NamedTuple())
        F = first(scalar_free_energy(s, gc, ps, NamedTuple()))
        t == steps && break
        G = free_energy_gradient(s, gc, ps, NamedTuple(); ad = AutoForwardDiff()).mix
        grad = [G.sum.c; reduce(vcat, [[vec(G[Symbol(:sw, l)].b); G[Symbol(:sw, l)].logw] for l in 1:L]; init = Float64[])] ./ N
        θ = [c; reduce(vcat, [[vec(sws[l].b); sws[l].logw] for l in 1:L]; init = Float64[])]
        vel = vel === nothing ? zero(θ) : vel
        vel = γ .* vel .- η .* grad
        θ = θ .+ vel
        c = θ[1:d]
        o = d
        sws = map(1:L) do l
            b = reshape(θ[(o + 1):(o + 2d)], d, 2); w = θ[(o + 2d + 1):(o + 2d + 2)]; o += 2d + 2
            (A = (u = E,), b = b, logw = w)
        end
    end
    return c, sws, F, s
end
# Fit by EM instead (`STRUCTURE_FIT=em`): closed-form M-steps from the cluster beliefs, SQUAREM-
# accelerated (`Mycelium.fit_em`), until F falls by less than 1e-4 (BIC to 1e-3). The frozen coefficients stay.
function fit_by_em(gc, L, N, c, sws; s0 = nothing, iters = 300, tol = 1e-4)
    ps = full_ps(gc, make_ps(L, c, sws), N)
    ps, Fs, s = fit_em(gc, ps, NamedTuple(); schedule = plate_schedule(gc), iters, tol, accelerate = true, store = s0)
    return ps.mix.sum.c, Any[ps.mix[Symbol(:sw, l)] for l in 1:L], Fs[end], s
end
const FIT = get(ENV, "STRUCTURE_FIT", "gradient")
fit(args...; kwargs...) = FIT == "em" ? fit_by_em(args...; kwargs...) : fit_gradient(args...; kwargs...)
bic(F, L, N) = 2F + (d + L * (d + 1)) * log(N)

# data: E3's 8-mode grid in 2-d, z = c + Σₗ kₗ Wₗ + ε, three switches
rng = Xoshiro(3)
const Wtrue = [2.0 0.0 1.0; 0.0 2.0 -1.0] .* 1.5
const N = 200
X = reduce(hcat, [Wtrue * rand(rng, Bool, 3) .+ [-1.0, 0.5] .+ sqrt(0.1) .* randn(rng, d) for _ in 1:N])

g = base_graph(X)
L, c, sws = 0, vec(mean(X; dims = 2)), Any[]
gc = clustered(g, L, N)
c, sws, F, s = fit(gc, L, N, c, sws)
best = bic(F, L, N)
@printf("start      L = %d  F = %9.2f  BIC = %9.2f\n", L, F, best)
path = [(L, best)]
t0 = time()
while true
    cands = []
    # add a switch: warm start with the current fit and a new switch of small random offset (4 draws)
    gA = add_switch(g, L, N); gcA = clustered(gA, L + 1, N)
    for r in 1:4
        new = (A = (u = E,), b = hcat(zeros(d), 0.5 .* randn(rng, d)), logw = zeros(2))
        cA, swA, FA, sA = fit(gcA, L + 1, N, c, [sws; new]; s0 = transfer(s, gc, gcA))
        push!(cands, (:add, bic(FA, L + 1, N), gA, gcA, L + 1, cA, swA, sA))
    end
    # remove the least used switch (the most unequal weights)
    if L > 0
        use = [minimum(exp.(sw.logw .- log(sum(exp.(sw.logw))))) for sw in sws]
        l = argmin(use)
        gR = remove_switch(g, L, N, l); gcR = clustered(gR, L - 1, N)
        swR = [sws[j] for j in 1:L if j != l]
        l < L && (swR = [swR[1:(l - 1)]; [sws[L]]; swR[l:end - 1]])
        cR, swR, FR, sR = fit(gcR, L - 1, N, c, swR)
        push!(cands, (:remove, bic(FR, L - 1, N), gR, gcR, L - 1, cR, swR, sR))
    end
    i = argmin([x[2] for x in cands])
    move, score, gN, gcN, LN, cN, swN, sN = cands[i]
    score < best - 1e-6 || break
    global g, gc, L, c, sws, s, best = gN, gcN, LN, cN, swN, sN, score
    push!(path, (L, best))
    @printf("%-6s ->  L = %d  BIC = %9.2f\n", move, L, best)
end
@printf("greedy search: L = %d in %.1f s; path %s\n", L, time() - t0, join(["L=$(p[1])" for p in path], " → "))

# reference: every L ≤ 4 from random starts (4 each), the same fit
println()
@printf("%3s %12s\n", "L", "best BIC")
for Lr in 0:4
    gr = base_graph(X)
    for l in 0:(Lr - 1); gr = add_switch(gr, l, N); end
    gcr = clustered(gr, Lr, N)
    bests = minimum(1:4) do r
        swr = [(A = (u = E,), b = hcat(zeros(d), 2 .* randn(rng, d)), logw = zeros(2)) for _ in 1:Lr]
        _, _, Fr, _ = fit(gcr, Lr, N, vec(mean(X; dims = 2)), swr)
        bic(Fr, Lr, N)
    end
    @printf("%3d %12.2f\n", Lr, bests)
end
