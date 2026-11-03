# Greedy structure search over E3b's switch over sums, (J clusters) × (L switches), with all four of
# the branch's moves (`experiment-qm`, examples/structure_switches.jl) done as edits of one Mycelium
# graph and scored by its exact free energy. Extends `examples/structure_edits.jl` (J = 1) to J > 1.
# Recorded in `examples/structure_clusters.md`.
# Run:  julia -t auto --project=docs examples/structure_clusters.jl   (an environment with ForwardDiff)
#       STRUCTURE_GREEDY=false skips the greedy search and runs the reference only.
#
# The model: z = c_j + Σₗ kₗ W_{jl} + ε, j ~ π, kₗ | j ~ Bernoulli(p_{jl}). As a graph, per point:
#   - the label j (J states) read by a switch `sw0` whose outcome sets u₀ = c_j (log-weights log π);
#   - per switch l an outcome qₗ = (bit, j) with 2J states, read by a switch setting uₗ = 0 or W_{jl}, and a
#     `TableFactor` over (j, qₗ) that is log J when qₗ's cluster is j and -Inf otherwise (it ties qₗ to j);
#   - the sum z − u₀ − Σₗ uₗ with noise 0.1·I, and the data.
# The switch's log-weights are log p(bit | j) − log J, normalised over its 2J outcomes, so with the
# table's log J the joint is exactly the model's. Every point is one cluster node (tied, `params = :mix`),
# so the graph's free energy is −Σᵢ log p(zᵢ), with the impossible assignments dropped inside.
#
# Moves (edits of the unclustered graph): add a switch (every sum node gains a channel), remove the
# least used switch, split the heaviest cluster, merge the two closest clusters (both change the
# number of states of j and of every qₗ, so those variables are re-added and their factors replaced).
# Fit: Momentum on the free-energy gradient at the exact beliefs (generalised EM), on c, log π, W and
# the logits of p. Score: BIC = 2F + p log N, p = J(d + 1 + L(d + 1)) − 1.
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random, Printf, Statistics
using ADTypes, DifferentiationInterface, ForwardDiff

const d = 2
const E = Matrix(1.0I, d, d)
nm(s, i) = Symbol(s, "_", i)
chans(ids) = (:z, :u0, (Symbol(:u, s) for s in ids)...)
sumfactor(ids) = LinearConstraintFactor(NamedTuple{chans(ids)}(ntuple(_ -> d, length(ids) + 2)), d; noise = 0.1)
switch(K) = SwitchingConstraintFactor((u = d,), K, d; noise = 1e-3)
function table(J)
    T = fill(-Inf, J, 2J)
    for j in 1:J; T[j, 2j - 1] = T[j, 2j] = log(J); end
    return TableFactor((j = J, q = 2J), T)
end

# --- the graph and its edits -----------------------------------------------------------------------
function base_graph(X, J)
    b = GraphBuilder()
    for i in axes(X, 2)
        z, j, u0 = nm(:z, i), nm(:j, i), nm(:u0, i)
        variable!(b, z, d); variable!(b, j, J); variable!(b, u0, d)
        factor!(b, nm(:data, i), DataFactor(:z, X[:, i])); connect!(b, nm(:data, i), :z, z; direction = Emitting())
        factor!(b, nm(:sum, i), sumfactor(Int[]); params = :sum)
        connect!(b, nm(:sum, i), :z, z); connect!(b, nm(:sum, i), :u0, u0)
        factor!(b, nm(:sw0, i), switch(J); params = :sw0); connect!(b, nm(:sw0, i), :u, u0); connect!(b, nm(:sw0, i), :k, j)
    end
    return build(b)
end
function add_switch(g, J, ids, s, N)
    b = GraphBuilder(g)
    for i in 1:N
        replace_factor!(b, nm(:sum, i), sumfactor([ids; s]))
        u, q = nm(Symbol(:u, s), i), nm(Symbol(:q, s), i)
        sw, t = nm(Symbol(:sw, s), i), nm(Symbol(:t, s), i)
        variable!(b, u, d); variable!(b, q, 2J)
        factor!(b, sw, switch(2J); params = Symbol(:sw, s)); factor!(b, t, table(J); params = Symbol(:t, s))
        connect!(b, nm(:sum, i), Symbol(:u, s), u); connect!(b, sw, :u, u); connect!(b, sw, :k, q)
        connect!(b, t, :j, nm(:j, i)); connect!(b, t, :q, q)
    end
    return build(b)
end
function remove_switch(g, ids, s, N)
    b = GraphBuilder(g)
    for i in 1:N
        remove_factor!(b, nm(Symbol(:sw, s), i)); remove_factor!(b, nm(Symbol(:t, s), i))
        remove_variable!(b, nm(Symbol(:u, s), i)); remove_variable!(b, nm(Symbol(:q, s), i))
        replace_factor!(b, nm(:sum, i), sumfactor(setdiff(ids, [s])))   # keeps the other channels' edges
    end
    return build(b)
end
function resize_clusters(g, Jnew, ids, N)          # split or merge: j and every qₗ change their states
    b = GraphBuilder(g)
    for i in 1:N
        j = nm(:j, i)
        remove_variable!(b, j); variable!(b, j, Jnew)
        replace_factor!(b, nm(:sw0, i), switch(Jnew)); connect!(b, nm(:sw0, i), :k, j)
        for s in ids
            q, sw, t = nm(Symbol(:q, s), i), nm(Symbol(:sw, s), i), nm(Symbol(:t, s), i)
            remove_variable!(b, q); variable!(b, q, 2Jnew)
            replace_factor!(b, sw, switch(2Jnew)); connect!(b, sw, :k, q)
            replace_factor!(b, t, table(Jnew)); connect!(b, t, :j, j); connect!(b, t, :q, q)
        end
    end
    return build(b)
end
function clustered(g, ids, N)
    inner(i) = [nm(:sum, i); nm(:sw0, i); [nm(Symbol(:sw, s), i) for s in ids]; [nm(Symbol(:t, s), i) for s in ids]]
    return cluster(g, [nm(:c, i) => inner(i) for i in 1:N]; params = :mix)
end

# --- parameters ------------------------------------------------------------------------------------
# θ = (a: log π up to a constant, C: d×J centres, W: d×J×L offsets, r: J×L logits of p)
logσ(x) = -log1p(exp(-x))
function switch_ps(W, r, l)
    J = size(W, 2)
    B = zeros(d, 2J); lw = zeros(2J)
    for j in 1:J
        B[:, 2j] = W[:, j, l]
        lw[2j - 1] = logσ(-r[j, l]) - log(J); lw[2j] = logσ(r[j, l]) - log(J)
    end
    return (A = (u = E,), b = B, logw = lw)
end
function make_ps(θ, ids)
    L = length(ids)
    sum_ps = (A = NamedTuple{chans(ids)}((E, ntuple(_ -> -E, L + 1)...)), c = zeros(d))
    return (; sum = sum_ps, sw0 = (A = (u = E,), b = θ.C, logw = θ.a),
            (Symbol(:sw, ids[l]) => switch_ps(θ.W, θ.r, l) for l in 1:L)...,
            (Symbol(:t, s) => NamedTuple() for s in ids)...)
end
full_ps(inner, N) = merge(NamedTuple{Tuple(nm(:data, i) for i in 1:N)}(ntuple(_ -> NamedTuple(), N)), (mix = inner,))
nparams(J, L) = J * (d + 1 + L * (d + 1)) - 1
bic(F, J, L, N) = 2F + nparams(J, L) * log(N)

# the gradient in θ from the gradient in the graph's parameters (chain rule through `switch_ps`)
function θgrad(G, θ, ids)
    J, L = size(θ.C, 2), length(ids)
    dW, dr = zeros(d, J, L), zeros(J, L)
    for l in 1:L
        Gs = G[Symbol(:sw, ids[l])]
        for j in 1:J
            dW[:, j, l] = Gs.b[:, 2j]
            dr[j, l] = Gs.logw[2j] * exp(logσ(-θ.r[j, l])) - Gs.logw[2j - 1] * exp(logσ(θ.r[j, l]))
        end
    end
    return (a = collect(G.sw0.logw), C = Matrix(G.sw0.b), W = dW, r = dr)
end
vecθ(θ) = [θ.a; vec(θ.C); vec(θ.W); vec(θ.r)]
function unvec(v, J, L)
    o = 0
    take(n) = (x = v[(o + 1):(o + n)]; o += n; x)
    a = take(J); C = reshape(take(d * J), d, J); W = reshape(take(d * J * L), d, J, L); r = reshape(take(J * L), J, L)
    return (a = a, C = C, W = W, r = r)
end

function fit(gc, ids, N, θ; steps = 300, η = 0.02, γ = 0.9, s0 = nothing)
    J, L = size(θ.C, 2), length(ids)
    sched = SequentialSchedule(all_tasks(gc))
    s = s0 === nothing ? MessageStore(gc) : s0
    vel = zeros(length(vecθ(θ)))
    F = Inf
    for t in 1:steps
        ps = full_ps(make_ps(θ, ids), N)
        sweep!(s, gc, sched, ps, NamedTuple()); sweep!(s, gc, sched, ps, NamedTuple())
        F = first(scalar_free_energy(s, gc, ps, NamedTuple()))
        t == steps && break
        G = free_energy_gradient(s, gc, ps, NamedTuple(); ad = AutoForwardDiff()).mix
        vel = γ .* vel .- η .* vecθ(θgrad(G, θ, ids)) ./ N
        θ = unvec(vecθ(θ) .+ vel, J, L)
    end
    return θ, F, s
end

# --- warm starts: the branch's four moves ------------------------------------------------------------
const S = sqrt(0.1)       # the noise scale, sqrt(tr Σ / d)
weights(θ) = exp.(θ.a .- maximum(θ.a)) ./ sum(exp.(θ.a .- maximum(θ.a)))
add_θ(θ, rng) = (a = θ.a, C = θ.C, W = cat(θ.W, reshape(2S .* randn(rng, d, size(θ.C, 2)), d, :, 1); dims = 3),
                 r = hcat(θ.r, zeros(size(θ.C, 2))))
function least_used(θ)
    p = 1 ./ (1 .+ exp.(-θ.r))
    return argmin(vec(mean(min.(p, 1 .- p); dims = 1)))
end
drop_θ(θ, l) = (a = θ.a, C = θ.C, W = θ.W[:, :, setdiff(1:end, l)], r = θ.r[:, setdiff(1:end, l)])
function split_θ(θ, rng)
    w = weights(θ); j = argmax(w)
    C = hcat(θ.C, θ.C[:, j] .+ S .* randn(rng, d)); C[:, j] .-= S .* randn(rng, d)
    lw = log.([w; w[j] / 2]); lw[j] = log(w[j] / 2)
    return (a = lw, C = C, W = cat(θ.W, θ.W[:, j:j, :]; dims = 2), r = vcat(θ.r, θ.r[j:j, :]))
end
function merge_θ(θ)
    J = size(θ.C, 2); w = weights(θ)
    a, b = argmin(((a, b),) -> norm(θ.C[:, a] - θ.C[:, b]), [(a, b) for a in 1:J for b in (a + 1):J])
    keep = setdiff(1:J, (a, b)); wa, wb = w[a], w[b]
    avg(x, y) = (wa .* x .+ wb .* y) ./ (wa + wb)
    p = 1 ./ (1 .+ exp.(-θ.r)); pm = avg(p[a, :], p[b, :])
    return (a = log.([w[keep]; wa + wb]), C = hcat(θ.C[:, keep], avg(θ.C[:, a], θ.C[:, b])),
            W = cat(θ.W[:, keep, :], reshape(avg(θ.W[:, a, :], θ.W[:, b, :]), d, 1, :); dims = 2),
            r = vcat(θ.r[keep, :], reshape(log.(pm ./ (1 .- pm)), 1, :)))
end

# --- data: E3b's tree, 3 clusters of rotated 2 × 2 grids (the branch's `tree_data`), true (3, 2) -----
rot(a) = [cos(a) -sin(a); sin(a) cos(a)]
function tree_data(N, rng)
    centres = [5.0 .* [cos(a), sin(a)] for a in (π / 2, π / 2 + 2π / 3, π / 2 + 4π / 3)]
    angles, probs, wts = (0.0, 0.5, 1.1), ([0.5, 0.4], [0.3, 0.6], [0.7, 0.5]), [0.4, 0.35, 0.25]
    Σh = cholesky([0.12 0.04; 0.04 0.09]).L
    return reduce(hcat, map(1:N) do _
        j = findfirst(cumsum(wts) .>= rand(rng))
        R = rot(angles[j])
        centres[j] .+ (rand(rng) < probs[j][1]) .* (R * [1.8, 0.0]) .+ (rand(rng) < probs[j][2]) .* (R * [0.0, 1.4]) .+ Σh * randn(rng, d)
    end)
end

rng = Xoshiro(5)
const N = 200
X = tree_data(N, rng)

if get(ENV, "STRUCTURE_GREEDY", "true") != "false"
J, ids, next = 1, Int[], 1
g = base_graph(X, J)
gc = clustered(g, ids, N)
θ = (a = [0.0], C = reshape(vec(mean(X; dims = 2)), d, 1), W = zeros(d, 1, 0), r = zeros(1, 0))
θ, F, s = fit(gc, ids, N, θ)
best = bic(F, J, 0, N)
@printf("start              (J, L) = (1, 0)  BIC = %9.2f\n", best)
path = ["(1, 0)"]
nfits = 1
t0 = time()
while true
    cands = []
    L = length(ids)
    if L < 3                                                        # add a switch (4 draws)
        gA = add_switch(g, J, ids, next, N); idsA = [ids; next]; gcA = clustered(gA, idsA, N)
        for _ in 1:4
            θA, FA, sA = fit(gcA, idsA, N, add_θ(θ, rng); s0 = transfer(s, gc, gcA)); global nfits += 1
            push!(cands, ("add a switch", bic(FA, J, L + 1, N), gA, gcA, J, idsA, θA, sA, next + 1))
        end
    end
    if L > 0                                                        # remove the least used switch
        l = least_used(θ); idsR = ids[setdiff(1:L, l)]
        gR = remove_switch(g, ids, ids[l], N); gcR = clustered(gR, idsR, N)
        θR, FR, sR = fit(gcR, idsR, N, drop_θ(θ, l); s0 = transfer(s, gc, gcR)); global nfits += 1
        push!(cands, ("remove a switch", bic(FR, J, L - 1, N), gR, gcR, J, idsR, θR, sR, next))
    end
    if J < 4                                                        # split the heaviest cluster (4 draws)
        gS = resize_clusters(g, J + 1, ids, N); gcS = clustered(gS, ids, N)
        for _ in 1:4
            θS, FS, sS = fit(gcS, ids, N, split_θ(θ, rng); s0 = transfer(s, gc, gcS)); global nfits += 1
            push!(cands, ("split a cluster", bic(FS, J + 1, L, N), gS, gcS, J + 1, ids, θS, sS, next))
        end
    end
    if J > 1                                                        # merge the two closest clusters
        gM = resize_clusters(g, J - 1, ids, N); gcM = clustered(gM, ids, N)
        θM, FM, sM = fit(gcM, ids, N, merge_θ(θ); s0 = transfer(s, gc, gcM)); global nfits += 1
        push!(cands, ("merge two clusters", bic(FM, J - 1, L, N), gM, gcM, J - 1, ids, θM, sM, next))
    end
    i = argmin([c[2] for c in cands])
    move, score, gN, gcN, JN, idsN, θN, sN, nextN = cands[i]
    score < best - 1e-6 || break
    global g, gc, J, ids, θ, s, best, next = gN, gcN, JN, idsN, θN, sN, score, nextN
    push!(path, "($J, $(length(ids)))")
    @printf("%-18s (J, L) = (%d, %d)  BIC = %9.2f   (%.0f s)\n", move, J, length(ids), best, time() - t0)
end
@printf("greedy search: (J, L) = (%d, %d) with %d fits in %.0f s; path %s\n", J, length(ids), nfits, time() - t0, join(path, " → "))
println("weights ", round.(weights(θ); digits = 3), "   centres ", round.(θ.C; digits = 2))
end

# reference: the generating structure (3, 2), fitted from the generating parameters and from 4 random
# starts (centres at random data points), its graph built by the same edits
println()
gT = base_graph(X, 1)
gT = add_switch(gT, 1, Int[], 1, N); gT = add_switch(gT, 1, [1], 2, N)
gT = resize_clusters(gT, 3, [1, 2], N)
gcT = clustered(gT, [1, 2], N)
angs = (0.0, 0.5, 1.1)
θtrue = (a = log.([0.4, 0.35, 0.25]), C = reduce(hcat, [5.0 .* [cos(a), sin(a)] for a in (π / 2, π / 2 + 2π / 3, π / 2 + 4π / 3)]),
         W = cat(reduce(hcat, [rot(angs[j]) * [1.8, 0.0] for j in 1:3]), reduce(hcat, [rot(angs[j]) * [0.0, 1.4] for j in 1:3]); dims = 3),
         r = log.([0.5 0.4; 0.3 0.6; 0.7 0.5] ./ (1 .- [0.5 0.4; 0.3 0.6; 0.7 0.5])))
_, FT, _ = fit(gcT, [1, 2], N, θtrue)
@printf("(3, 2) from the generating parameters: BIC = %9.2f\n", bic(FT, 3, 2, N))
for r in 1:4
    θr = (a = zeros(3), C = X[:, rand(rng, 1:N, 3)], W = 2S .* randn(rng, d, 3, 2), r = zeros(3, 2))
    _, Fr, _ = fit(gcT, [1, 2], N, θr)
    @printf("(3, 2) from random start %d:          BIC = %9.2f\n", r, bic(Fr, 3, 2, N))
end
