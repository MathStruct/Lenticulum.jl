"""
Structure learning for switch wirings: `vault/Design/Minimal Stochastic Factors.md` §10, and
`vault/Factor Graphs/Dynamic Graphs.md` (structural changes). The first rule here that changes a
graph's structure rather than its parameters.

The model family is E3b's switch over sums, indexed by its structure (J, L): J clusters, each an
additive model with L binary switches. L = 0 is a flat J-component mixture with a shared
covariance, and J = 1 is E3's single sum node. Four moves change the structure:

- **add a switch** (L → L + 1): every cluster gains an offset, warm-started at a small random
  vector with p = 1/2;
- **remove a switch** (L → L − 1): the least used switch (p nearest 0 or 1) is dropped;
- **split a cluster** (J → J + 1): the heaviest cluster is duplicated with perturbed centres;
- **merge two clusters** (J → J − 1): the two closest centres are averaged.

Each candidate is refitted by exact EM from its warm start and from random starts, and scored by
BIC on the training data. Greedy search starts at (1, 0) and takes the best improving move until
none improves. An exhaustive search over the grid J ≤ 9, L ≤ 4 is the reference. Graphs are not
mutated: each candidate is a new model, as in `examples/gaussian_structure_search.jl`.
Run with `julia -t auto --project=. examples/structure_switches.jl`.
"""
module StructureSwitches

using LinearAlgebra, Random, Printf

isdefined(Main, :HierarchyE3b) || Base.include(Main, joinpath(@__DIR__, "hierarchy_e3b.jl"))
const H = Main.HierarchyE3b
const C3 = Main.CompositionE3
const E = Main.SwitchingE4a

mean(v) = sum(v) / length(v)

"The data sets, with their true structure (J, L)."
datasets() = (
    grid3 = (C3.DATA.grid3, (1, 3)),
    grid4 = (C3.DATA.grid4, (1, 4)),
    tree = (H.tree_data(), (3, 2)),
    ring = (C3.DATA.ring, (8, 0)),
)

bic(ll, n, J, L) = -2n * ll + H.nhier(J, L) * log(n)

# --- warm starts for the four moves ---------------------------------------------------------------

function add_switch(h::H.Hier, rng)
    s = sqrt(tr(h.Σ) / 2)
    H.Hier(copy(h.π), [hcat(W, 2s .* randn(rng, 2)) for W in h.W], [vcat(p, 0.5) for p in h.p], copy(h.Σ))
end
function remove_switch(h::H.Hier)
    L = length(h.p[1]); L == 0 && return nothing
    use = [mean(min(p[l], 1 - p[l]) for p in h.p) for l in 1:L]
    l = argmin(use); keep = [1; [m + 1 for m in 1:L if m != l]]
    H.Hier(copy(h.π), [W[:, keep] for W in h.W], [p[setdiff(1:L, l)] for p in h.p], copy(h.Σ))
end
function split_cluster(h::H.Hier, rng)
    j = argmax(h.π); s = sqrt(tr(h.Σ) / 2)
    W2 = copy(h.W[j]); W2[:, 1] .+= s .* randn(rng, 2)
    W1 = copy(h.W[j]); W1[:, 1] .-= s .* randn(rng, 2)
    π = [h.π; h.π[j] / 2]; π[j] /= 2
    H.Hier(π, [h.W[1:j-1]; [W1]; h.W[j+1:end]; [W2]], [h.p; [copy(h.p[j])]], copy(h.Σ))
end
function merge_clusters(h::H.Hier)
    J = length(h.π); J == 1 && return nothing
    pairs = [(a, b) for a in 1:J for b in a+1:J]
    a, b = argmin(((a, b),) -> norm(h.W[a][:, 1] - h.W[b][:, 1]), pairs)
    wa, wb = h.π[a], h.π[b]
    W = (wa .* h.W[a] .+ wb .* h.W[b]) ./ (wa + wb)
    p = (wa .* h.p[a] .+ wb .* h.p[b]) ./ (wa + wb)
    keep = setdiff(1:J, (a, b))
    H.Hier([h.π[keep]; wa + wb], [h.W[keep]; [W]], [h.p[keep]; [p]], copy(h.Σ))
end

"Fit structure (J, L) by exact EM from the warm starts given and `starts` random starts; the best by training likelihood."
function fit_structure(J, L, X; warm = H.Hier[], starts = 32, seed = 1)
    inits = [warm; [H.hier_init(J, L, X, Xoshiro(seed + 7919s + 101J + L)) for s in 1:starts]]
    res = fetch.([Threads.@spawn H.em_hier(h0, X) for h0 in inits])
    lls = [mean(H.hier_logpdf(r[1], X)) for r in res]
    b = argmax(lls)
    return res[b][1], lls[b]
end

"Greedy search from (1, 0): the best improving move by BIC, until none improves."
function greedy(X; starts = 32, maxJ = 9, maxL = 4, seed = 1)
    n = size(X, 1); rng = Xoshiro(seed)
    J, L = 1, 0
    h, ll = fit_structure(J, L, X; starts, seed)
    score = bic(ll, n, J, L)
    path = [(; J, L, score, move = "start")]; fits = 1
    while true
        cands = []
        L < maxL && push!(cands, ("add a switch", J, L + 1, [add_switch(h, rng) for _ in 1:4]))
        L > 0 && push!(cands, ("remove a switch", J, L - 1, [remove_switch(h)]))
        J < maxJ && push!(cands, ("split a cluster", J + 1, L, [split_cluster(h, rng) for _ in 1:4]))
        J > 1 && push!(cands, ("merge two clusters", J - 1, L, [merge_clusters(h)]))
        best = nothing
        for (move, Jc, Lc, warm) in cands
            hc, llc = fit_structure(Jc, Lc, X; warm, starts, seed); fits += 1
            sc = bic(llc, n, Jc, Lc)
            (best === nothing || sc < best.score) && (best = (; move, J = Jc, L = Lc, h = hc, score = sc))
        end
        (best === nothing || best.score >= score - 1e-6) && break
        J, L, h, score = best.J, best.L, best.h, best.score
        push!(path, (; J, L, score, move = best.move))
    end
    return (; J, L, h, score, path, fits)
end

"Every structure on the grid, by BIC."
function exhaustive(X; starts = 32, maxJ = 9, maxL = 4, seed = 1)
    n = size(X, 1)
    grid = [(J, L) for J in 1:maxJ for L in 0:maxL]
    scores = Dict((J, L) => bic(last(fit_structure(J, L, X; starts, seed)), n, J, L) for (J, L) in grid)
    return argmin(scores), scores
end

function run(; ntrain = 2000, ntest = 2000, starts = 32, maxJ = 9, maxL = 4, names = (:grid3, :grid4, :tree, :ring))
    rows = []
    for name in names
        d, true_s = datasets()[name]
        X = E.sample_gmm(d.π, d.μ, d.Σ, ntrain, Xoshiro(10))
        Xte = E.sample_gmm(d.π, d.μ, d.Σ, ntest, Xoshiro(11))
        truth = mean([E.gmm_logpdf(d.π, d.μ, d.Σ, Xte[i, :]) for i in axes(Xte, 1)])
        tg = @elapsed g = greedy(X; starts, maxJ, maxL)
        te = @elapsed (ex, scores) = exhaustive(X; starts, maxJ, maxL)
        gap = truth - mean(H.hier_logpdf(g.h, Xte))
        path = join(["$(p.move == "start" ? "" : p.move * " → ")($(p.J), $(p.L))" for p in g.path], ", ")
        @printf("%s: true structure %s; greedy finds %s with %d fits (%.1f s), exhaustive BIC optimum %s over %d structures (%.1f s)\n",
                name, true_s, (g.J, g.L), g.fits, tg, ex, length(scores), te)
        @printf("  greedy path: %s\n  held-out gap of the greedy model %.3f; BIC of the true structure %.1f, of the greedy one %.1f\n",
                path, gap, scores[true_s], g.score)
        push!(rows, (; name, true_s, found = (g.J, g.L), exhaustive = ex, fits = g.fits, nstruct = length(scores), gap,
                     path = g.path, bic_true = scores[true_s], bic_found = g.score))
    end
    return rows
end

main() = run()

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    StructureSwitches.main()
end
