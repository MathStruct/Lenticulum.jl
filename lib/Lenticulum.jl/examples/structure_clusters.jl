# Greedy structure search over E3b's switch over sums, (J clusters) × (L switches), with all four of
# the branch's moves (`experiment-qm`, examples/structure_switches.jl) done as edits of one Mycelium
# graph and scored by its exact free energy. Extends `examples/structure_edits.jl` (J = 1) to J > 1.
# Recorded in `examples/structure_clusters.md`.
# Run:  julia -t auto --project=docs lib/Lenticulum.jl/examples/structure_clusters.jl   (an environment with ForwardDiff)
#       STRUCTURE_GREEDY=false skips the greedy search and runs the reference only;
#       STRUCTURE_COMPOUND=false leaves out the compound moves and the data-driven splits (the branch's
#       four moves only); STRUCTURE_FIT=em fits by EM in θ instead of 300 gradient steps;
#       STRUCTURE_BEAM=k (k > 1) runs a beam search of width k instead of the greedy one, followed by a
#       refit of the chosen structure from STRUCTURE_REFIT starts (32 with EM). The fastest route to the
#       generating structure's best fit: STRUCTURE_FIT=em STRUCTURE_BEAM=2 (about 12 minutes).
#
# The model: z = c_j + Σₗ kₗ W_{jl} + ε, j ~ π, kₗ | j ~ Bernoulli(p_{jl}). As a graph, per point:
#   - the label j (J states) read by a switch `sw0` whose outcome sets u₀ = c_j (log-weights log π);
#   - per switch l an outcome qₗ = (bit, j) with 2J states, read by a switch setting uₗ = 0 or W_{jl}, and a
#     `TableFactor` over (j, qₗ) that is log J when qₗ's cluster is j and -Inf otherwise (it ties qₗ to j);
#   - the sum z − u₀ − Σₗ uₗ with noise 0.1·I, and the data.
# The switch's log-weights are log p(bit | j) − log J, normalised over its 2J outcomes, so with the
# table's log J the joint is exactly the model's. Every point is one cluster node (tied, `params = :mix`),
# so the graph's free energy is −Σᵢ log p(zᵢ), with the impossible assignments dropped inside. The
# points are recorded as a plate (`as_plate` after each edit), so the clusters' messages and free
# energies are batched.
#
# Moves (edits of the unclustered graph): add a switch (every sum node gains a channel), remove the
# least used switch, split the heaviest cluster, merge the two closest clusters (both change the
# number of states of j and of every qₗ, so those variables are re-added and their factors replaced),
# and two compound moves, a split with a removal and a merge with an addition: no single move leads
# from (2, 3) to (3, 2), where the four-move search stops on the tree. Splits also get a data-driven
# warm start (2-means on the heaviest cluster's points), and the compound split tries every switch.
# Fit: Momentum on the free-energy gradient at the exact beliefs (generalised EM), on c, log π, W and
# the logits of p. Score: BIC = 2F + p log N, p = J(d + 1 + L(d + 1)) − 1.
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random, Printf, Statistics
using MyceliumLearning
using ADTypes, DifferentiationInterface, ForwardDiff

const d = 2
const E = Matrix(1.0I, d, d)
nm(s, i) = replica_name(s, i)
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
        factor!(b, nm(:sum, i), sumfactor(Int[]); params = :sum, freeze = (:A,))
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
# every point's component as one cluster node, the points a plate (batched messages and energies)
clustered(g, ids, N) = cluster(as_plate(g, :pts, N), :pts, :c,
                               [:sum; :sw0; [Symbol(:sw, s) for s in ids]; [Symbol(:t, s) for s in ids]]; params = :mix)

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

function fit_gradient(gc, ids, N, θ; steps = 300, η = 0.02, γ = 0.9, s0 = nothing)
    J, L = size(θ.C, 2), length(ids)
    sched = plate_schedule(gc)
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

# θ from the graph's parameters (the inverse of `make_ps`)
function θ_of(inner, ids)
    J, L = size(inner.sw0.b, 2), length(ids)
    W, r = zeros(d, J, L), zeros(J, L)
    for l in 1:L
        sw = inner[Symbol(:sw, ids[l])]
        for j in 1:J
            W[:, j, l] = sw.b[:, 2j]
            r[j, l] = sw.logw[2j] - sw.logw[2j - 1]
        end
    end
    return (a = collect(inner.sw0.logw), C = Matrix(inner.sw0.b), W = W, r = r)
end
# EM in θ (`STRUCTURE_FIT=em`), collapsed: given the assignment a of a point's discrete variables
# (its label j and the bits of its switches), the u's integrate out and z ~ N(c_j + Σₗ bitₗ W_jl, R)
# with a fixed R, so the M-step for (c_j, W_j·) is one weighted least-squares fit per cluster over
# the assignments' posterior weights, read from each point's cluster belief; π_j and the logits
# r_jl are the posterior masses. This is EM on the same likelihood with fewer latent variables.
# The per-factor M-step (`MyceliumLearning.em_step` on the switches) crawls here: a switch of noise 1e-3
# pins its u to the current offset, and each step moves it by little (`examples/structure_clusters.md`).
function em_θ(s, gc, ps, st, ids, N, X)
    θ = θ_of(ps.mix, ids)
    J, L = size(θ.C, 2), length(ids)
    nj, n1 = zeros(J), zeros(J, L)
    G = [zeros(L + 1, L + 1) for _ in 1:J]; R = [zeros(L + 1, d) for _ in 1:J]
    for i in 1:N
        fid = gc.factor_index[nm(:c, i)]
        fn = Mycelium.factornode(gc, fid)
        c = fn.factor
        b = Mycelium.factor_belief(c, Mycelium.messages_into(s, gc, fid), Mycelium._subtree(ps, fn.params), NamedTuple())
        ij = findfirst(==(:j), c.discrete)
        iq = [findfirst(==(Symbol(:q, t)), c.discrete) for t in ids]
        z = X[:, i]
        for (a, qa) in zip(b.assignments, b.q)
            qa > 0 || continue
            j = a[ij]
            φ = [1.0; [a[iq[l]] == 2j ? 1.0 : 0.0 for l in 1:L]]     # state 2j of q is "bit on" in cluster j
            nj[j] += qa; n1[j, :] .+= qa .* φ[2:end]
            G[j] .+= qa .* (φ * φ'); R[j] .+= qa .* (φ * z')
        end
    end
    a = log.(max.(nj, floatmin(Float64)) ./ N)
    C, W, r = copy(θ.C), copy(θ.W), copy(θ.r)
    for j in 1:J
        nj[j] > 1e-12 || continue
        B = pinv(G[j]) * R[j]                                    # (L + 1) × d
        C[:, j] = B[1, :]
        for l in 1:L
            n1[j, l] > 1e-12 && (W[:, j, l] = B[1 + l, :])
            r[j, l] = log(max(n1[j, l], floatmin(Float64))) - log(max(nj[j] - n1[j, l], floatmin(Float64)))
        end
    end
    return full_ps(make_ps((a = a, C = C, W = W, r = r), ids), N)
end
function fit_by_em_once(gc, ids, N, θ; s0 = nothing, iters = 300, tol = 1e-4)
    ps, Fs, s = fit_em(gc, full_ps(make_ps(θ, ids), N), NamedTuple(); schedule = plate_schedule(gc), iters, tol,
                       accelerate = true, store = s0, mstep = (s, g, p, st) -> em_θ(s, g, p, st, ids, N, X))
    return θ_of(ps.mix, ids), Fs[end], s
end
# Restarts (`STRUCTURE_RESTARTS`, default 4): EM settles on stationary points the gradient fit escapes
# (at (1, 0), three of four add-a-switch draws ended at BIC 31 942.95 by EM, one by gradient steps), so
# each EM fit also starts from perturbed copies of its warm start (centres by S/2, offsets by S,
# logits by 1; a separate generator, so the search's own draws are unchanged) and keeps the best.
const RESTARTS = parse(Int, get(ENV, "STRUCTURE_RESTARTS", "4"))
const RESTART_RNG = Xoshiro(2026)
perturb(θ, rng) = (a = θ.a, C = θ.C .+ (S / 2) .* randn(rng, size(θ.C)), W = θ.W .+ S .* randn(rng, size(θ.W)),
                   r = θ.r .+ randn(rng, size(θ.r)))
function fit_by_em(gc, ids, N, θ; s0 = nothing, kwargs...)
    best = fit_by_em_once(gc, ids, N, θ; s0 = s0 === nothing ? nothing : deepcopy(s0), kwargs...)
    for _ in 2:RESTARTS
        cand = fit_by_em_once(gc, ids, N, perturb(θ, RESTART_RNG); s0 = s0 === nothing ? nothing : deepcopy(s0), kwargs...)
        cand[2] < best[2] && (best = cand)
    end
    return best
end
const FIT = get(ENV, "STRUCTURE_FIT", "gradient")
fit(args...; kwargs...) = FIT == "em" ? fit_by_em(args...; kwargs...) : fit_gradient(args...; kwargs...)

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
# A data-driven split: the heaviest cluster's points (responsibility > 1/2, read from each point's
# cluster node) are split by 2-means; each half's centre is its mean minus the switches' expected
# offsets, its weight the cluster's share by size.
function data_split_θ(θ, s, gc, ids, X; j = argmax(weights(θ)))
    N = size(X, 2); w = weights(θ)
    ps = full_ps(make_ps(θ, ids), N)
    resp = [exp(cluster_marginal(s, gc, nm(:c, i), :j, ps).logp[j]) for i in 1:N]
    P = X[:, resp .> 0.5]
    size(P, 2) >= 2 || return split_θ(θ, Xoshiro(1))
    a, b = Tuple(argmax(((p, q),) -> norm(P[:, p] - P[:, q]), [(p, q) for p in axes(P, 2) for q in (p + 1):size(P, 2)]))
    μ = hcat(P[:, a], P[:, b])
    lab = ones(Int, size(P, 2))
    for _ in 1:20
        lab = [argmin([norm(P[:, t] - μ[:, k]) for k in 1:2]) for t in axes(P, 2)]
        for k in 1:2; any(==(k), lab) && (μ[:, k] = vec(mean(P[:, lab .== k]; dims = 2))); end
    end
    p = 1 ./ (1 .+ exp.(-θ.r[j, :]))
    shift = isempty(p) ? zeros(d) : sum(p[l] .* θ.W[:, j, l] for l in eachindex(p))
    f = count(==(1), lab) / length(lab)
    C = hcat(θ.C, μ[:, 2] .- shift); C[:, j] = μ[:, 1] .- shift
    lw = log.([w; w[j] * (1 - f)]); lw[j] = log(w[j] * f)
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

const COMPOUND = get(ENV, "STRUCTURE_COMPOUND", "true") != "false"
rng = Xoshiro(5)
const N = 200
X = tree_data(N, rng)

# every move's fitted candidates from one state: (move, BIC, g, gc, J, ids, θ, store, next)
function candidates(g, gc, J, ids, θ, s, next)
    nf = 0
    cands = []
    L = length(ids)
    if L < 3                                                        # add a switch (4 draws)
        gA = add_switch(g, J, ids, next, N); idsA = [ids; next]; gcA = clustered(gA, idsA, N)
        for _ in 1:4
            θA, FA, sA = fit(gcA, idsA, N, add_θ(θ, rng); s0 = transfer(s, gc, gcA)); nf += 1
            push!(cands, ("add a switch", bic(FA, J, L + 1, N), gA, gcA, J, idsA, θA, sA, next + 1))
        end
    end
    if L > 0                                                        # remove the least used switch
        l = least_used(θ); idsR = ids[setdiff(1:L, l)]
        gR = remove_switch(g, ids, ids[l], N); gcR = clustered(gR, idsR, N)
        θR, FR, sR = fit(gcR, idsR, N, drop_θ(θ, l); s0 = transfer(s, gc, gcR)); nf += 1
        push!(cands, ("remove a switch", bic(FR, J, L - 1, N), gR, gcR, J, idsR, θR, sR, next))
    end
    if J < 4                                                        # split the heaviest cluster (4 draws)
        gS = resize_clusters(g, J + 1, ids, N); gcS = clustered(gS, ids, N)
        for _ in 1:4
            θS, FS, sS = fit(gcS, ids, N, split_θ(θ, rng); s0 = transfer(s, gc, gcS)); nf += 1
            push!(cands, ("split a cluster", bic(FS, J + 1, L, N), gS, gcS, J + 1, ids, θS, sS, next))
        end
        if COMPOUND
            θS, FS, sS = fit(gcS, ids, N, data_split_θ(θ, s, gc, ids, X); s0 = transfer(s, gc, gcS)); nf += 1
            push!(cands, ("split a cluster", bic(FS, J + 1, L, N), gS, gcS, J + 1, ids, θS, sS, next))
        end
    end
    if J > 1                                                        # merge the two closest clusters
        gM = resize_clusters(g, J - 1, ids, N); gcM = clustered(gM, ids, N)
        θM, FM, sM = fit(gcM, ids, N, merge_θ(θ); s0 = transfer(s, gc, gcM)); nf += 1
        push!(cands, ("merge two clusters", bic(FM, J - 1, L, N), gM, gcM, J - 1, ids, θM, sM, next))
    end
    if COMPOUND && J < 4 && L > 0                                   # a data-driven split, then remove each switch in turn
        θD = data_split_θ(θ, s, gc, ids, X)
        for l in 1:L
            idsC = ids[setdiff(1:L, l)]
            gC = remove_switch(resize_clusters(g, J + 1, ids, N), ids, ids[l], N); gcC = clustered(gC, idsC, N)
            θC, FC, sC = fit(gcC, idsC, N, drop_θ(θD, l); s0 = transfer(s, gc, gcC)); nf += 1
            push!(cands, ("split, remove a switch", bic(FC, J + 1, L - 1, N), gC, gcC, J + 1, idsC, θC, sC, next))
        end
    end
    if COMPOUND && J > 1 && L < 3                                   # merge, then add a switch
        gC = add_switch(resize_clusters(g, J - 1, ids, N), J - 1, ids, next, N); idsC = [ids; next]; gcC = clustered(gC, idsC, N)
        for _ in 1:4
            θC, FC, sC = fit(gcC, idsC, N, add_θ(merge_θ(θ), rng); s0 = transfer(s, gc, gcC)); nf += 1
            push!(cands, ("merge, add a switch", bic(FC, J - 1, L + 1, N), gC, gcC, J - 1, idsC, θC, sC, next + 1))
        end
    end
    return cands, nf
end

const BEAM = parse(Int, get(ENV, "STRUCTURE_BEAM", "1"))
if get(ENV, "STRUCTURE_GREEDY", "true") != "false" && BEAM == 1
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
    cands, nf = candidates(g, gc, J, ids, θ, s, next); global nfits += nf
    i = argmin([c[2] for c in cands])
    move, score, gN, gcN, JN, idsN, θN, sN, nextN = cands[i]
    if !(score < best - 1e-6)       # the stop: the best candidate of each move
        for mv in unique(c[1] for c in cands)
            @printf("  stop: best %-22s (J, L) = (%d, %d)  BIC = %9.2f\n", mv, first(c[5] for c in cands if c[1] == mv),
                    first(length(c[6]) for c in cands if c[1] == mv), minimum(c[2] for c in cands if c[1] == mv))
        end
        break
    end
    global g, gc, J, ids, θ, s, best, next = gN, gcN, JN, idsN, θN, sN, score, nextN
    push!(path, "($J, $(length(ids)))")
    @printf("%-18s (J, L) = (%d, %d)  BIC = %9.2f   (%.0f s)\n", move, J, length(ids), best, time() - t0)
end
@printf("greedy search: (J, L) = (%d, %d) with %d fits in %.0f s; path %s\n", J, length(ids), nfits, time() - t0, join(path, " → "))
println("weights ", round.(weights(θ); digits = 3), "   centres ", round.(θ.C; digits = 2))
end

# beam search (`STRUCTURE_BEAM=k`, k > 1): every kept structure is expanded by all moves, the best fit
# per structure (J, L) not expanded before is kept, and the k best of those are the next beam; the
# search stops when no candidate improves on the best score seen. k = 1 is the greedy search above.
if get(ENV, "STRUCTURE_GREEDY", "true") != "false" && BEAM > 1
gB = base_graph(X, 1); gcB = clustered(gB, Int[], N)
θB, FB, sB = fit(gcB, Int[], N, (a = [0.0], C = reshape(vec(mean(X; dims = 2)), d, 1), W = zeros(d, 1, 0), r = zeros(1, 0)))
start = (g = gB, gc = gcB, J = 1, ids = Int[], θ = θB, s = sB, next = 1, score = bic(FB, 1, 0, N), path = ["(1, 0)"])
@printf("beam %d: start (J, L) = (1, 0)  BIC = %9.2f\n", BEAM, start.score)
beam, bestst, seen, nfitsB, tB = [start], start, Set([(1, 0)]), 1, time()
while true
    pool = Dict{Tuple{Int,Int},Any}()
    for st in beam
        cands, nf = candidates(st.g, st.gc, st.J, st.ids, st.θ, st.s, st.next); global nfitsB += nf
        for c in cands
            key = (c[5], length(c[6]))
            key in seen && continue
            (haskey(pool, key) && pool[key][1][2] <= c[2]) || (pool[key] = (c, st.path))
        end
    end
    isempty(pool) && break
    ranked = sort(collect(values(pool)); by = x -> x[1][2])
    ranked[1][1][2] < bestst.score - 1e-6 || break
    global beam = map(ranked[1:min(BEAM, length(ranked))]) do (c, path)
        move, score, gN, gcN, JN, idsN, θN, sN, nextN = c
        (g = gN, gc = gcN, J = JN, ids = idsN, θ = θN, s = sN, next = nextN, score = score, path = [path; "($JN, $(length(idsN)))"])
    end
    foreach(b -> push!(seen, (b.J, length(b.ids))), beam)
    beam[1].score < bestst.score && (global bestst = beam[1])
    @printf("beam: %s   (%.0f s)\n", join([@sprintf("(%d, %d) %.2f", b.J, length(b.ids), b.score) for b in beam], ";  "), time() - tB)
end
@printf("beam search (k = %d): (J, L) = (%d, %d), BIC = %.2f with %d fits in %.0f s; path %s\n", BEAM, bestst.J,
        length(bestst.ids), bestst.score, nfitsB, time() - tB, join(bestst.path, " → "))
# a final refit of the chosen structure (`STRUCTURE_REFIT=n`, default 32 with EM): half the starts
# perturb its fit, half are random (centres at random data points), as the reference's; the best is kept
nrefit = parse(Int, get(ENV, "STRUCTURE_REFIT", FIT == "em" ? "32" : "0"))
if nrefit > 0
    tR = time()
    Jb, Lb = bestst.J, length(bestst.ids)
    starts = [isodd(r) ? perturb(bestst.θ, RESTART_RNG) :
              (a = zeros(Jb), C = X[:, rand(RESTART_RNG, 1:N, Jb)], W = 2S .* randn(RESTART_RNG, d, Jb, Lb), r = zeros(Jb, Lb)) for r in 1:nrefit]
    fits = [fit_by_em_once(bestst.gc, bestst.ids, N, θr) for θr in starts]
    scores = [bic(f[2], Jb, Lb, N) for f in fits]
    k = argmin(scores)
    if scores[k] < bestst.score
        global bestst = merge(bestst, (θ = fits[k][1], s = fits[k][3], score = scores[k]))
    end
    @printf("refit (%d starts, %s): best BIC %.2f (from a %s start), median %.2f; %.0f s\n", nrefit, FIT, scores[k],
            isodd(k) ? "perturbed" : "random", sort(scores)[cld(nrefit, 2)], time() - tR)
end
println("weights ", round.(weights(bestst.θ); digits = 3), "   centres ", round.(bestst.θ.C; digits = 2))
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
