"""
E7: structured inference inside E3b's hierarchy, `vault/Design/Minimal Stochastic Factors.md`
§10, suggested by the second round of reviews (Grok; Gemini's "E7"). The model is E3b's switch
over sums in d dimensions:

    j ~ Categorical(w),  k_l | j ~ Bernoulli(p_{jl}),  z = c_j + Σ_l k_l δ_{jl} + ε,  ε ~ N(0, Σ_j),

with J clusters of L switches, J·2^L components. E6 showed that factorised posteriors over the
switches bias learning through the pair moments E[k_l k_m]. Here every E-step keeps the top switch
exact and approximates q(k | j) by a product over *blocks* of switches, exact within a block:

- **exact**: one block of all L switches (the oracle, J·2^L states per point);
- **mean field**: blocks of one switch;
- **fixed blocks**: blocks of B consecutive switches;
- **overlap blocks**: blocks of at most B switches, chosen per cluster at every E-step by merging
  the switches whose offsets overlap most (largest |cos| in the noise metric Σ_j⁻¹).

Within a block the posterior and its pair moments are exact; across blocks the pair moments
factorise, E[k_l k_m] = q_l q_m. The block updates are coordinate ascent on the evidence bound
(structured mean field, as in Ghahramani & Jordan 1997), and q(j) ∝ exp(ELBO_j). The
M-step is closed form per cluster. Every model is scored by its exact held-out log-likelihood and
by the energy distance of generated samples.

Run with `julia -t auto --project=. examples/hierarchy_inference.jl`.
"""
module HierarchyInference

using LinearAlgebra, Random, Printf

# --- the model -----------------------------------------------------------------------------------

"Cluster j: W_j = [c_j δ_{j1} … δ_{jL}] (d × (L+1)), bit probabilities p_j, noise Σ_j; weights w."
struct Hier
    w::Vector{Float64}; W::Vector{Matrix{Float64}}; p::Vector{Vector{Float64}}; Σ::Vector{Matrix{Float64}}
end
nclusters(h::Hier) = length(h.w)
nswitches(h::Hier) = length(h.p[1])
nparams(J, L, d) = (J - 1) + J * (d * (L + 1) + L + d * (d + 1) ÷ 2)

"J clusters 6 apart in random directions; offsets of length 2.5 in random directions."
function truth(d, J, L; seed = 1)
    rng = Xoshiro(seed)
    W = [hcat(6.0 .* normalize(randn(rng, d)), [2.5 .* normalize(randn(rng, d)) for _ in 1:L]...) for _ in 1:J]
    p = [0.3 .+ 0.4 .* rand(rng, L) for _ in 1:J]
    Σ = [Matrix((0.2 + 0.1rand(rng)) * I, d, d) for _ in 1:J]
    w = normalize(1 .+ rand(rng, J), 1)
    return Hier(w, W, p, Σ)
end

function sample(h::Hier, n, rng)
    d = size(h.W[1], 1); L = nswitches(h)
    X = zeros(n, d)
    cw = cumsum(h.w)
    for i in 1:n
        j = min(searchsortedfirst(cw, rand(rng)), nclusters(h))
        k = rand(rng, L) .< h.p[j]
        X[i, :] = h.W[j] * vcat(1.0, k) + cholesky(Symmetric(h.Σ[j])).L * randn(rng, d)
    end
    return X
end

configs(B) = reduce(hcat, [digits(i; base = 2, pad = B) for i in 0:2^B-1])    # B × 2^B

# --- the E-step: exact top switch, block-factorised switches ---------------------------------------

"Blocks of at most B switches that merge the most overlapping pairs first (cos in the Σ⁻¹ metric)."
function overlap_blocks(G, B)
    L = size(G, 1)
    blocks = [[l] for l in 1:L]
    c = [abs(G[l, m]) / sqrt(G[l, l] * G[m, m]) for l in 1:L, m in 1:L]
    while true
        best, pair = 0.0, nothing
        for a in eachindex(blocks), b in a+1:length(blocks)
            length(blocks[a]) + length(blocks[b]) <= B || continue
            s = maximum(c[l, m] for l in blocks[a] for m in blocks[b])
            s > best && ((best, pair) = (s, (a, b)))
        end
        pair === nothing && break
        a, b = pair
        blocks[a] = sort(vcat(blocks[a], blocks[b])); deleteat!(blocks, b)
    end
    return blocks
end
fixed_blocks(L, B) = [collect(s:min(s + B - 1, L)) for s in 1:B:L]

"""
For one point x and one cluster: structured mean field over `blocks`. Returns the evidence bound
ELBO_j (exact log p(x, j) for a single block), E[k] and E[k kᵀ] under q(k | j).
"""
function cluster_posterior(x, w, W, p, Σ, blocks, Kb; sweeps = 50, tol = 1e-9)
    d, L = size(W, 1), length(p)
    C = cholesky(Symmetric(Σ)); Δ = W[:, 2:end]; r = x - W[:, 1]
    Λr = C \ r; h = Δ' * Λr; G = Δ' * (C \ Δ)
    lp1, lp0 = log.(p), log1p.(-p)
    m = copy(p)
    qs = Vector{Vector{Float64}}(undef, length(blocks))
    for sweep in 1:sweeps
        change = 0.0
        for (bi, b) in enumerate(blocks)
            K = Kb[length(b)]
            other = setdiff(1:L, b)
            hb = h[b] - G[b, other] * m[other]
            s = vec(K' * (hb .+ lp1[b] .- lp0[b])) .- 0.5 .* vec(sum(K .* (G[b, b] * K); dims = 1))
            s .-= maximum(s); q = exp.(s); q ./= sum(q)
            qs[bi] = q
            mb = K * q
            change = max(change, maximum(abs.(mb - m[b])))
            m[b] = mb
        end
        (change < tol || length(blocks) == 1) && break
    end
    # second moments: exact within a block, factorised across
    M2 = m * m'
    elbo = log(w) - (d * log(2π) + logdet(C)) / 2 - dot(r, Λr) / 2 + sum(lp0)
    for (bi, b) in enumerate(blocks)
        K = Kb[length(b)]; q = qs[bi]
        M2[b, b] = K * (q .* K')
        lin = vec(K' * (h[b] .+ lp1[b] .- lp0[b])) .- 0.5 .* vec(sum(K .* (G[b, b] * K); dims = 1))
        elbo += dot(q, lin) - sum(qi * log(qi) for qi in q if qi > 0)
    end
    for (a, ba) in enumerate(blocks), (c, bc) in enumerate(blocks)
        a == c || (elbo -= 0.5 * dot(m[ba], G[ba, bc] * m[bc]))
    end
    return elbo, m, M2
end

"Sufficient statistics of the whole E-step: per cluster Σ q_ij, Σ q_ij E[φ], E[φφᵀ], x E[φ]ᵀ."
function estep(h::Hier, X, blocking)
    n, d = size(X); J, L = nclusters(h), nswitches(h)
    Kb = [configs(B) for B in 1:L]
    Gs = [h.W[j][:, 2:end]' * (Symmetric(h.Σ[j]) \ h.W[j][:, 2:end]) for j in 1:J]
    blocks = [blocking(Gs[j]) for j in 1:J]
    per = Vector{Any}(undef, n)
    Threads.@threads for i in 1:n
        x = X[i, :]
        res = [cluster_posterior(x, h.w[j], h.W[j], h.p[j], h.Σ[j], blocks[j], Kb) for j in 1:J]
        e = [r[1] for r in res]; mx = maximum(e); qj = exp.(e .- mx); qj ./= sum(qj)
        per[i] = (qj, res, mx + log(sum(exp.(e .- mx))))
    end
    st = map(1:J) do j
        Sφφ = zeros(L + 1, L + 1); Sxφ = zeros(d, L + 1); Sxx = zeros(d, d); N = 0.0
        for i in 1:n
            q = per[i][1][j]; _, m, M2 = per[i][2][j]; x = X[i, :]
            Eφ = vcat(1.0, m); Eφφ = [1.0 m'; m M2]
            N += q; Sφφ .+= q .* Eφφ; Sxφ .+= q .* (x * Eφ'); Sxx .+= q .* (x * x')
        end
        (; N, Sφφ, Sxφ, Sxx)
    end
    return st, sum(p[3] for p in per) / n, blocks
end

"Closed-form M-step per cluster."
function mstep(st, n; floor = 1e-6)
    W = [s.Sxφ / Symmetric(s.Sφφ + 1e-10I) for s in st]
    Σ = [Matrix(Symmetric((s.Sxx - Wj * s.Sxφ' - s.Sxφ * Wj' + Wj * s.Sφφ * Wj') / max(s.N, 1e-12))) + floor * I
         for (s, Wj) in zip(st, W)]
    p = [clamp.(s.Sφφ[1, 2:end] ./ max(s.N, 1e-12), 1e-6, 1 - 1e-6) for s in st]
    w = [max(s.N, 1e-12) / n for s in st]
    return Hier(w ./ sum(w), W, p, Σ)
end

"Exact mean log-likelihood (one block of all switches)."
exact_loglik(h::Hier, X) = estep(h, X, G -> [collect(1:size(G, 1))])[2]

"EM with the given blocking until the parameters stop moving."
function fit(h::Hier, X, blocking; iters = 200, tol = 1e-6)
    for _ in 1:iters
        hn = mstep(first(estep(h, X, blocking)), size(X, 1))
        Δ = maximum(maximum(abs.(a - b)) for (a, b) in zip(hn.W, h.W))
        Δ = max(Δ, maximum(maximum(abs.(a - b)) for (a, b) in zip(hn.Σ, h.Σ)), maximum(abs.(hn.w - h.w)))
        h = hn
        Δ < tol && break
    end
    return h
end

"A random start: clusters at random data points, offsets as differences of data points."
function init(J, L, X, rng)
    d = size(X, 2); pick() = X[rand(rng, axes(X, 1)), :]
    v = minimum(eigvals(Symmetric(cov(X)))) / 4
    return Hier(fill(1 / J, J), [hcat(pick(), [0.5 .* (pick() - pick()) for _ in 1:L]...) for _ in 1:J],
                [fill(0.5, L) for _ in 1:J], [Matrix(v * I, d, d) for _ in 1:J])
end
cov(X) = (D = X .- sum(X; dims = 1) / size(X, 1); D' * D / size(X, 1))

function energy_distance(X, Y)
    dist(A, B) = sum(sqrt(max(sum(abs2, A[i, :] .- B[j, :]), 0.0)) for i in axes(A, 1), j in axes(B, 1)) / (size(A, 1) * size(B, 1))
    return 2dist(X, Y) - dist(X, X) - dist(Y, Y)
end

# --- the experiment ------------------------------------------------------------------------------

methods(L, B) = (
    ("exact", G -> [collect(1:size(G, 1))]),
    ("mean field", G -> [[l] for l in 1:size(G, 1)]),
    ("fixed blocks of $B", G -> fixed_blocks(size(G, 1), B)),
    ("overlap blocks of ≤ $B", G -> overlap_blocks(G, B)),
)

"""
For each truth seed and L (blocks of B = L/2): every E-step's EM fixed point from the true
parameters, and for the seeds in `random_seeds` the best of `starts` random starts (by the exact
training likelihood). Scores: the exact held-out gap and the energy distance of `ngen` samples.
"""
function run(; d = 8, J = 3, Ls = (6, 8), seeds = 1:3, random_seeds = (1,), ntrain = 4000, ntest = 2000,
             ngen = 1000, starts = 16, iters = 200)
    rows = []
    for seed in seeds, L in Ls
        B = L ÷ 2
        tm = truth(d, J, L; seed)
        X = sample(tm, ntrain, Xoshiro(10 + 100seed)); Xte = sample(tm, ntest, Xoshiro(11 + 100seed))
        t_ll = exact_loglik(tm, Xte)
        floor_ed = energy_distance(sample(tm, ngen, Xoshiro(12 + 100seed)), Xte[1:ngen, :])
        @printf("seed %d, d = %d, J = %d, L = %d (%d components, %d parameters): truth %.3f; energy-distance floor %.4f\n",
                seed, d, J, L, J * 2^L, nparams(J, L, d), t_ll, floor_ed)
        for (name, blocking) in methods(L, B)
            t1 = @elapsed hf = fit(tm, X, blocking; iters)
            gap = t_ll - exact_loglik(hf, Xte)
            ed = energy_distance(sample(hf, ngen, Xoshiro(13)), Xte[1:ngen, :])
            gr, tr = NaN, NaN
            if seed in random_seeds
                tr = @elapsed fits = [fit(init(J, L, X, Xoshiro(s)), X, blocking; iters) for s in 1:starts]
                best = fits[argmax([exact_loglik(f, X) for f in fits])]
                gr = t_ll - exact_loglik(best, Xte)
            end
            push!(rows, (; seed, L, B, method = name, gap, ed, floor_ed, gap_random = gr, time = t1, time_random = tr))
            @printf("  %-24s from the truth: gap %.3f, energy distance %.4f (%.1f s)%s\n", name, gap, ed, t1,
                    isnan(gr) ? "" : @sprintf("   best of %d random starts: gap %.3f (%.1f s)", starts, gr, tr))
        end
    end
    return rows
end

main() = run()

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    HierarchyInference.main()
end
