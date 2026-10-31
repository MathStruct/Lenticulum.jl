"""
E3b of `vault/Design/Minimal Stochastic Factors.md` (§8, §10): a deeper wiring. A top-level switch
chooses one of J additive models (E3's sum node fed by L binary switches), so the components are
arranged as a tree: J clusters, each a 2^L grid with its own offsets. This is a switch over sums,
the next wiring after E3's single sum.

Data: 3 clusters at the corners of a triangle, each a 2 × 2 grid rotated by its own angle (12 modes,
one shared covariance). It is neither a flat product (the local grids are rotated differently) nor
unstructured.

Models, all fitted by exact EM from random starts on parallel threads:

- **hierarchical** (J = 3, L = 2): p(z) = Σ_j π_j Σ_k Π_l p_jl^{k_l}(1 − p_jl)^{1−k_l} N(z; c_j + Σ_l k_l δ_jl, Σ),
  J(3L + 3) + 2 parameters (29); its posterior over (j, k) is exact and the M-step is closed form
  (a weighted least squares per cluster);
- **additive** (one sum node, L ∈ {2, 3, 4}): E3's `em_additive`;
- **flat**: K ∈ {3, 6, 12}, 6K − 1 parameters.

Run with `julia -t auto --project=. examples/hierarchy_e3b.jl`.
"""
module HierarchyE3b

using LinearAlgebra, Random, Printf

isdefined(Main, :CompositionE3) || Base.include(Main, joinpath(@__DIR__, "composition_e3.jl"))
const C3 = Main.CompositionE3
const E = Main.SwitchingE4a

# --- data ----------------------------------------------------------------------------------------

function tree_data()
    centres = [5.0 .* [cos(a), sin(a)] for a in (π / 2, π / 2 + 2π / 3, π / 2 + 4π / 3)]
    angles = (0.0, 0.5, 1.1)
    probs = ([0.5, 0.4], [0.3, 0.6], [0.7, 0.5])
    weights = [0.4, 0.35, 0.25]
    π_, μ = Float64[], Vector{Float64}[]
    for j in 1:3
        R = C3.rot(angles[j]); δ = [R * [1.8, 0.0], R * [0.0, 1.4]]
        for k in Iterators.product(0:1, 0:1)
            push!(μ, centres[j] + k[1] * δ[1] + k[2] * δ[2])
            push!(π_, weights[j] * prod(k[l] == 1 ? probs[j][l] : 1 - probs[j][l] for l in 1:2))
        end
    end
    return (; π = π_, μ, Σ = [C3.SHARED for _ in μ])
end

# --- the hierarchical model: a switch over additive models ----------------------------------------

"Parameters: π (J), per cluster W_j = [c_j δ_j1 … δ_jL] (2 × (L+1)) and bit probabilities p_j (L), shared Σ."
struct Hier
    π::Vector{Float64}; W::Vector{Matrix{Float64}}; p::Vector{Vector{Float64}}; Σ::Matrix{Float64}
end
nhier(J, L) = J * (3L + 3) + 2

function hier_loglik_matrix(h::Hier, Xt, Φ, Kc)
    C = cholesky(Symmetric(h.Σ)); U = C.L \ Xt; uu = sum(abs2, U; dims = 1)'
    blocks = map(eachindex(h.π)) do j
        M = h.W[j] * Φ; V = C.L \ M
        lw = vec(log.(h.p[j])' * Kc .+ log1p.(-h.p[j])' * (1 .- Kc)) .+ log(h.π[j])
        lw' .- (2log(2π) + logdet(C)) / 2 .- (uu .- 2 * (U' * V) .+ sum(abs2, V; dims = 1)) ./ 2
    end
    return reduce(hcat, blocks)                                       # n × (J 2^L)
end

function hier_logpdf(h::Hier, X)
    L = length(h.p[1])
    cfgs = [digits(i; base = 2, pad = L) for i in 0:2^L-1]
    Φ = reduce(hcat, [[1.0; k] for k in cfgs]); Kc = reduce(hcat, cfgs)
    Lg = hier_loglik_matrix(h, Matrix(X'), Φ, Kc)
    mx = maximum(Lg; dims = 2)
    return vec(mx .+ log.(sum(exp.(Lg .- mx); dims = 2)))
end

function hier_init(J, L, X, rng)
    pick() = X[rand(rng, axes(X, 1)), :]
    s2 = minimum(eigvals(E.cov(X))) / 16
    W = [hcat(pick(), [0.5 .* randn(rng, 2) for _ in 1:L]...) for _ in 1:J]
    return Hier(fill(1 / J, J), W, [fill(0.5, L) for _ in 1:J], Matrix(s2 * I, 2, 2))
end

"Exact EM: the posterior over (j, k) is exact; the M-step is a weighted least squares per cluster."
function em_hier(h::Hier, X; iters = 500, tol = 1e-9, floor = 1e-6)
    n = size(X, 1); J = length(h.π); L = length(h.p[1]); nc = 2^L
    cfgs = [digits(i; base = 2, pad = L) for i in 0:nc-1]
    Φ = reduce(hcat, [[1.0; k] for k in cfgs]); Kc = reduce(hcat, cfgs)
    Xt = Matrix(X'); lls = Float64[]
    for it in 1:iters
        Lg = hier_loglik_matrix(h, Xt, Φ, Kc)
        mx = maximum(Lg; dims = 2); s = sum(exp.(Lg .- mx); dims = 2)
        push!(lls, sum(mx .+ log.(s)) / n)
        it > 1 && lls[end] - lls[end-1] < tol && break
        R = exp.(Lg .- mx) ./ s
        W = similar(h.W); p = similar(h.p); πn = zeros(J)
        S = zeros(2, 2)
        for j in 1:J
            Rj = R[:, (j-1)*nc+1:j*nc]; Rk = vec(sum(Rj; dims = 1)); nj = sum(Rk)
            πn[j] = max(nj / n, 1e-12)
            A = Φ * Diagonal(Rk) * Φ' + 1e-9I
            W[j] = (Xt * Rj * Φ') / A
            M = W[j] * Φ
            S .+= Xt * Diagonal(vec(sum(Rj; dims = 2))) * Xt' - M * (Rj' * X) - (Xt * Rj) * M' + M * Diagonal(Rk) * M'
            p[j] = clamp.(Kc * Rk ./ max(nj, 1e-12), 1e-9, 1 - 1e-9)
        end
        h = Hier(πn, W, p, Matrix(Symmetric(S / n)) + floor * I)
    end
    return h, lls
end

mean(v) = sum(v) / length(v)

function run(; ntrains = (150, 2000), ntest = 2000, starts = 256, Ls = (2, 3, 4), Ks = (3, 6, 12))
    d = tree_data(); J, L = 3, 2
    Xte = E.sample_gmm(d.π, d.μ, d.Σ, ntest, Xoshiro(2))
    truth = mean([E.gmm_logpdf(d.π, d.μ, d.Σ, Xte[i, :]) for i in axes(Xte, 1)])
    sep = minimum(norm(a - b) for a in d.μ for b in d.μ if a !== b)
    @printf("tree data (%d modes, min separation %.2f): held-out log p of the truth %.3f\n", length(d.μ), sep, truth)
    rows = []
    for n in ntrains
        X = E.sample_gmm(d.π, d.μ, d.Σ, n, Xoshiro(10 + n))
        res = fetch.([Threads.@spawn em_hier(hier_init(J, L, X, Xoshiro(s)), X) for s in 1:starts])
        tr = [mean(hier_logpdf(r[1], X)) for r in res]
        best = argmax(tr)
        mono = all(all(diff(r[2]) .> -1e-9) for r in res)
        push!(rows, (; n, model = "hierarchical, J = $J, L = $L", params = nhier(J, L),
                     test = mean(hier_logpdf(res[best][1], Xte)), hit = count(>(tr[best] - 1e-3), tr), mono))
        for La in Ls
            ra = fetch.([Threads.@spawn C3.em_additive(La, X, C3.additive_init(La, X, Xoshiro(s))) for s in 1:starts])
            ta = [mean(C3.additive_logpdf(r[1], La, X)) for r in ra]
            b = argmax(ta)
            push!(rows, (; n, model = "additive, L = $La", params = C3.nadd(La),
                         test = mean(C3.additive_logpdf(ra[b][1], La, Xte)), hit = count(>(ta[b] - 1e-3), ta), mono = true))
        end
        for K in Ks
            g = E.em_gmm(X, K; iters = 200, restarts = 10, floor = 1e-3, seed = 3)
            push!(rows, (; n, model = "flat, K = $K", params = 6K - 1,
                         test = mean([E.gmm_logpdf(g..., Xte[i, :]) for i in axes(Xte, 1)]), hit = 0, mono = true))
        end
        for r in rows[end-length(Ls)-length(Ks):end]
            @printf("  n = %4d  %-28s %3d params   held-out log p %.3f   (gap %.3f)%s\n", r.n, r.model, r.params,
                    r.test, truth - r.test, r.hit > 0 ? "   $(r.hit)/$starts starts at the best" : "")
        end
        @printf("  hierarchical EM log-likelihood monotone in every start: %s\n", rows[end-length(Ls)-length(Ks)].mono)
    end
    return (; truth, rows)
end

main() = run()

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    HierarchyE3b.main()
end
