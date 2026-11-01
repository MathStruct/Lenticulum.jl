"""
The objective, not the class: why the chains of E4(b) score like flat K-component mixtures.
`vault/Design/Minimal Stochastic Factors.md` §8, Propositions 9 and 10.

Proposition 10: a chain of L kernels with two outcomes each, constant gates, G = I, shifts
h_{l,k} = k δ_l, small noise C = εI and the prior N(c, Σ − LεI) has the marginal of E3's additive
model, a 2^L-component mixture. So chains with K outcomes per step are not limited to K components
as a class. Proposition 9: with a fixed forward process, the chain's evidence bound decomposes
exactly into T per-step conditional KLs, so training it makes every step match the data's own
noised reverse conditional, which needs the data's own number of linear pieces.

This script checks both on a 16-mode product grid (E3's `grid4`, standardised to unit scale):

1. the additive model fitted by exact EM, rewritten as a 4-step chain of 2-outcome kernels; its
   marginal by path enumeration equals the additive density to rounding;
2. ELBO-trained chains (E4(b)'s procedure) with K ∈ {2, 4} outcomes and T ∈ {5, 10, 20} steps,
   scored by the importance-weighted bound, against flat K-component mixtures;
3. `objective_control`: the two objectives in one family (constant gates), both scored exactly.

Run with `julia -t auto --project=. examples/chain_objective.jl`.
"""
module ChainObjective

using LinearAlgebra, Random, Printf

isdefined(Main, :SwitchingChainE4b) || Base.include(Main, joinpath(@__DIR__, "switching_chain_e4b.jl"))
isdefined(Main, :CompositionE3) || Base.include(Main, joinpath(@__DIR__, "composition_e3.jl"))
const B = Main.SwitchingChainE4b
const C3 = Main.CompositionE3
const E = Main.SwitchingE4a

"E3's `grid4` mixture, centred and scaled to unit average variance per coordinate."
function standard_grid()
    d = C3.DATA.grid4
    m = sum(d.π[j] * d.μ[j] for j in eachindex(d.π))
    V = sum(d.π[j] * (d.Σ[j] + (d.μ[j] - m) * (d.μ[j] - m)') for j in eachindex(d.π))
    s = 1 / sqrt(tr(V) / 2)
    return (; π = d.π, μ = [s * (μ - m) for μ in d.μ], Σ = [s^2 * S for S in d.Σ], L = d.L)
end

"Proposition 10: the additive model θ (parameterisation of `CompositionE3.additive_logpdf`) as a chain."
function additive_chain(θ, L; ε = 1e-4)
    c = θ[1:2]; δ = [θ[2l+1:2l+2] for l in 1:L]; ρ = θ[2L+3:3L+2]
    l11, l21, l22 = C3.chol2(θ[3L+3:3L+5]); Lc = [l11 0; l21 l22]; Σ = Lc * Lc'
    I2 = Matrix(1.0I, 2, 2)
    kernels = [E.Kernel([C3.logsig(-ρ[l]), C3.logsig(ρ[l])], [zeros(2), zeros(2)], [I2, I2],
                        [I2, I2], [zeros(2), δ[l]], [ε * I2, ε * I2]) for l in 1:L]
    prior = ([1.0], [c], [Σ - L * ε * I2])
    return kernels, prior
end

"""
The exact marginal of a chain whose gates do not depend on z_t, by enumerating the paths: along
each path z_0 is Gaussian, with mean and covariance pushed through the linear experts.
"""
function constant_gate_logpdf(kernels, prior, z)
    T = length(kernels)
    terms = Float64[]
    for path in Iterators.product((eachindex(k.logw) for k in kernels)...)
        lw = 0.0; m = prior[2][1]; V = prior[3][1]
        for t in T:-1:1
            k = kernels[t]; j = path[t]
            lw += k.logw[j] - E.logsumexp(k.logw)
            m = k.G[j] * m + k.h[j]; V = k.G[j] * V * k.G[j]' + k.C[j]
        end
        push!(terms, lw + E.logN(z, m, V))
    end
    return E.logsumexp(terms)
end

mean(v) = sum(v) / length(v)

# --- the objective-controlled comparison (second round of reviews) --------------------------------

"""
A step with constant gates, p(z_{t-1} | z_t) = Σ_k π_k N(G_k z_t + h_k, C_k), fitted by conditional
maximum likelihood on the pairs (rows z_{t-1}, z_t) of `X`: EM for a mixture of linear regressions
with constant weights, from `starts` random responsibilities. This is the diffusion bound's optimum
for that step within the constant-gate family (Proposition 9).
"""
function fit_constant_gate_step(X, K; iters = 300, starts = 8, seed = 1, floor = 1e-6)
    n = size(X, 1); Y = X[:, 1:2]; Z = hcat(X[:, 3:4], ones(n))           # regressors [z_t; 1]
    best, bestll = nothing, -Inf
    for s in 1:starts
        rng = Xoshiro(seed + 31s)
        R = rand(rng, n, K); R ./= sum(R; dims = 2)
        π, B, C = fill(1 / K, K), [zeros(3, 2) for _ in 1:K], [Matrix(1.0I, 2, 2) for _ in 1:K]
        ll = -Inf
        for it in 1:iters
            for k in 1:K
                w = R[:, k]; nk = sum(w); π[k] = max(nk / n, 1e-12)
                B[k] = (Z' * (w .* Z) + 1e-9I) \ (Z' * (w .* Y))
                E_ = Y - Z * B[k]
                C[k] = Matrix(Symmetric(E_' * (w .* E_) / nk)) + floor * I
            end
            Lg = [log(π[k]) + E.logN(Y[i, :], B[k]' * Z[i, :], C[k]) for i in 1:n, k in 1:K]
            mx = maximum(Lg; dims = 2); sm = sum(exp.(Lg .- mx); dims = 2)
            lln = sum(mx .+ log.(sm)) / n
            R = exp.(Lg .- mx) ./ sm
            abs(lln - ll) < 1e-9 && (ll = lln; break)
            ll = lln
        end
        ll > bestll && ((best, bestll) = ((copy(π), deepcopy(B), deepcopy(C)), ll))
    end
    π, B, C = best
    I2 = Matrix(1.0I, 2, 2)
    return E.Kernel(log.(π), [zeros(2) for _ in 1:K], [I2 for _ in 1:K], [Matrix(B[k][1:2, :]') for k in 1:K],
                    [B[k][3, :] for k in 1:K], C)
end

"""
The same generative family (constant gates), two objectives: exact marginal likelihood (the
additive model by EM, as a chain) against the diffusion bound with the fixed forward process (one
constant-gate projection per step). Both are scored by their exact held-out marginal, by path
enumeration, and by the bound.
"""
function objective_control(; ntrain = 4000, ntest = 1000, chains = [(4, 2), (10, 2), (5, 4)], em_starts = 256, S = 64)
    d = standard_grid(); L = d.L
    Ztr = E.sample_gmm(d.π, d.μ, d.Σ, ntrain, Xoshiro(1))
    Zte = E.sample_gmm(d.π, d.μ, d.Σ, ntest, Xoshiro(2))
    truth = mean([E.gmm_logpdf(d.π, d.μ, d.Σ, Zte[i, :]) for i in axes(Zte, 1)])
    @printf("Objective control on the standardised 16-mode grid: truth %.3f\n", truth)
    rows = []
    res = fetch.([Threads.@spawn C3.em_additive(L, Ztr, C3.additive_init(L, Ztr, Xoshiro(1 + 1000s))) for s in 1:em_starts])
    θ = res[argmax([mean(C3.additive_logpdf(r[1], L, Ztr)) for r in res])][1]
    ks, pr = additive_chain(θ, L)
    ll = mean([constant_gate_logpdf(ks, pr, Zte[i, :]) for i in axes(Zte, 1)])
    push!(rows, (; objective = "exact marginal likelihood (EM)", T = L, K = 2, exact = ll))
    @printf("  %-34s T = %2d, K = %d   exact held-out gap %.3f\n", rows[end].objective, L, 2, truth - ll)
    αT2 = 0.8^20
    for (T, K) in chains
        sch = E.schedule(T, 1 - αT2^(1 / T))
        kc = fetch.([Threads.@spawn fit_constant_gate_step(B.SS.step_pairs(sch, t, Ztr, Xoshiro(100t + K)), K; seed = 7t) for t in 1:T])
        a = sch.α[end]; ZT = a .* Ztr .+ sqrt(1 - a^2) .* randn(Xoshiro(1), size(Ztr)...)
        pc = ([1.0], [vec(sum(ZT; dims = 1)) / size(ZT, 1)], [E.cov(ZT)])
        ex = E.tmean(i -> constant_gate_logpdf(kc, pc, Zte[i, :]), ntest)
        iw = E.tmean(i -> B.iw_bound(kc, pc, sch, Zte[i, :], S, Xoshiro(10i)), ntest)
        push!(rows, (; objective = "diffusion bound (fixed forward)", T, K, exact = ex, iw))
        @printf("  %-34s T = %2d, K = %d   exact held-out gap %.3f   IW bound (S = %d) gap %.3f\n",
                rows[end].objective, T, K, truth - ex, S, truth - iw)
    end
    return (; truth, rows)
end

function run(; ntrain = 4000, ntest = 1000, chains = [(T, K) for T in (5, 10, 20) for K in (2, 4)],
             Ks_flat = (2, 4, 16), S = 64, iters = 120, em_starts = 256)
    d = standard_grid(); L = d.L
    Ztr = E.sample_gmm(d.π, d.μ, d.Σ, ntrain, Xoshiro(1))
    Zte = E.sample_gmm(d.π, d.μ, d.Σ, ntest, Xoshiro(2))
    truth = mean([E.gmm_logpdf(d.π, d.μ, d.Σ, Zte[i, :]) for i in axes(Zte, 1)])
    @printf("grid4, standardised: held-out log p of the truth %.3f\n", truth)
    rows = []
    # 1. the additive model by exact EM, and the same model as a chain of 2-outcome kernels
    res = fetch.([Threads.@spawn C3.em_additive(L, Ztr, C3.additive_init(L, Ztr, Xoshiro(1 + 1000s))) for s in 1:em_starts])
    θ = res[argmax([mean(C3.additive_logpdf(r[1], L, Ztr)) for r in res])][1]
    ll_add = mean(C3.additive_logpdf(θ, L, Zte))
    ks, pr = additive_chain(θ, L)
    ll_chain = [constant_gate_logpdf(ks, pr, Zte[i, :]) for i in axes(Zte, 1)]
    err = maximum(abs.(ll_chain .- C3.additive_logpdf(θ, L, Zte)))
    push!(rows, (; model = "additive by EM = chain, T = $L, K = 2, constant gates", params = C3.nadd(L), ll = ll_add))
    @printf("  %-52s %5d params   exact held-out log p %.3f (gap %.3f)\n", rows[end].model, rows[end].params, ll_add, truth - ll_add)
    @printf("  the same model as a chain: max |log p_chain − log p_additive| over the test points %.1e\n", err)
    # 2. flat mixtures and ELBO-trained chains
    for K in Ks_flat
        g = E.em_gmm(Ztr, K; iters = 200, restarts = 10, floor = 1e-4, seed = 3)
        ll = mean([E.gmm_logpdf(g..., Zte[i, :]) for i in axes(Zte, 1)])
        push!(rows, (; model = "flat, K = $K", params = 6K - 1, ll))
        @printf("  %-52s %5d params   exact held-out log p %.3f (gap %.3f)\n", rows[end].model, 6K - 1, ll, truth - ll)
    end
    αT2 = 0.8^20
    for (T, K) in chains
        sch = E.schedule(T, 1 - αT2^(1 / T))
        kc, pc = B.fit_chain(Ztr, sch, K; iters)
        iw = E.tmean(i -> B.iw_bound(kc, pc, sch, Zte[i, :], S, Xoshiro(10i)), ntest)
        push!(rows, (; model = "ELBO-trained chain, T = $T, K = $K", params = B.nchain(T, K), ll = iw))
        @printf("  %-52s %5d params   IW bound (S = %d) %.3f (gap %.3f)\n", rows[end].model, rows[end].params, S, iw, truth - iw)
    end
    return (; truth, err, rows)
end

main() = (run(), objective_control())

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    ChainObjective.main()
end
